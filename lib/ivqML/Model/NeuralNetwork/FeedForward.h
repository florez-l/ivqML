// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <initializer_list>
#include <vector>
#include <ivqML/Model/Base.h>



#include <algorithm>




namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class FeedForward
        : public ivqML::Model::Base< _TReal, _TNatural >
      {
      public:
        using TReal      = _TReal;
        using TNatural   = _TNatural;
        using Self       = FeedForward;
        using Superclass = ivqML::Model::Base< TReal, TNatural >;
        using TMat       = typename Superclass::TMat;
        using TCol       = typename Superclass::TCol;
        using TRow       = typename Superclass::TRow;
        using TMatMap    = typename Superclass::TMatMap;
        using TCMatMap   = typename Superclass::TCMatMap;
        using TColMap    = typename Superclass::TColMap;
        using TCColMap   = typename Superclass::TCColMap;
        using TRowMap    = typename Superclass::TRowMap;
        using TCRowMap   = typename Superclass::TCRowMap;

        using TActivation = int;

      public:
        FeedForward( )
          : Superclass( 0 )
          {
          }
        virtual ~FeedForward( )
          {
          }

        void set_input_layer(
          const TNatural& i, const TNatural& o, TActivation a
          )
          {
            this->m_N.clear( );
            this->m_W.clear( );
            this->m_B.clear( );
            this->m_A.clear( );

            this->m_N.push_back( i );
            this->m_N.push_back( o );
            this->m_A.push_back( a );
          }
        void set_input_layer(
          const TNatural& i, const TNatural& o, const std::string& a
          )
          {
            this->set_input_layer( i, o, aaaa );
          }
        void add_layer( const TNatural& o, TActivation a )
          {
            this->m_N.push_back( o );
            this->m_A.push_back( a );
          }
        void add_layer( const TNatural& o, const std::string& a )
          {
            this->add_layer( o, aaaa );
          }
        TNatural number_of_layers( ) const
          {
            return( this->m_W.size( ) );
          }
        const TNatural& input_size( const TNatural& l = 0 ) const
          {
            if( l < this->m_N.size( ) )
              return( this->m_N[ l ] );
            else
              return( 0 );
          }
        const TNatural& output_size( const TNatural& l = 0 ) const
          {
            if( this->m_N.size( ) > 0 )
            {
              if( l == 0 )
                return( this->m_N[ this->m_N.size( ) - 1 ] );
              else
              {
                TNatural i = l + 1;
                if( i < this->m_N.size( ) )
                  return( this->m_N[ i ] );
                else
                  return( 0 );
              }
            }
            else
              return( 0 );
          }

        TReal& operator[]( std::initializer_list< TNatural > i )
          {
          }
        const TReal& operator[]( std::initializer_list< TNatural > i ) const
          {
          }

        virtual void init( )
          {
            // Reserve space for all parameters
            TNatural L = this->m_N.size( ) - 1;
            TNatural N = 0;
            for( TNatural l = 0; l < L; ++l )
              N += ( this->m_N[ l ] + 1 ) * this->m_N[ l + 1 ];
            this->_resize( N );

            // Map parameters memory
            TReal* b = this->m_P;
            for( TNatural l = 0; l < L; ++l )
            {
              TNatural i = this->m_N[ l ];
              TNatural o = this->m_N[ l + 1 ];

              this->m_W.push_back( TMatMap( b, i, o ) );
              b += i * o;
              this->m_B.push_back( TRowMap( b, 1, o ) );
              b += o;
            } // end for
          }

        template< class _TX >
        TMat operator()( const Eigen::EigenBase< _TX >& X ) const
          {
            TNatural L = this->number_of_layers( );
            TNatural M = X.rows( );
            TNatural mN
              =
              *( std::max_element( this->m_N.begin( ) + 1, this->m_N.end( ) ) );
            TReal* Ab
              =
              reinterpret_cast< TReal* >(
                std::calloc( ( mN << 1 ) * M, sizeof( TReal ) )
                );
            TReal* Zb = Ab + ( mN * M );

            TMatMap( Ab, M, this->m_N[ 0 ] ) = X.template cast< TReal >( );
            for( TNatural l = 0; l < L; ++l )
            {
              TNatural i = this->m_N[ l ];
              TNatural o = this->m_N[ l + 1 ];

              TMatMap( Zb, M, o )
                =
                ( TMatMap( Ab, M, i ) * this->m_W[ l ] ).array( )
                +
                this->m_B[ l ];
              TMatMap( Ab, M, o ) = this->m_A[ l ]( TMatMap( Zb, M, o ) );
            } // end for

            TMat A = TMatMap( Ab, M, this->output_size( ) );
            std::free( Ab );
            return( A );
          }

        /**
         * TODO: This method has no sense in neural networks
         */
        template< class _TX, class _Ty >
        void fit(
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1 = 0, const TReal& L2 = 0
          )
          {
            /* TODO
               if( n == 0 || m != y.rows( ) )
               throw AssertionError( 'There is no closed solution for a logistic regression.' )
            */
          }

        template< class _TG, class _TX, class _Ty >
        TReal cost_gradient(
          Eigen::EigenBase< _TG >& G,
          const Eigen::EigenBase< _TX >& bX,
          const Eigen::EigenBase< _Ty >& by,
          const TReal& L1, const TReal& L2
          )
          {
          }

        template< class _TX, class _Ty >
        TReal cost(
          const Eigen::EigenBase< _TX >& X,
          const Eigen::EigenBase< _Ty >& y
          )
          {
          }

      protected:
        virtual void _to_stream( std::ostream& o ) const
          {
          }

      protected:
        std::vector< TNatural >    m_N;
        std::vector< TMatMap >     m_W;
        std::vector< TRowMap >     m_B;
        std::vector< TActivation > m_A;
      };
    } // end namespace
  } // end namespace
} // end namespace

// TODO: #include <ivqML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __ivqML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
