// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <functional>
#include <initializer_list>
#include <vector>
#include <ivqML/Model/Base.h>



#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <string>




namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal >
      class ActivationFactory
      {
      public:
        using TReal = _TReal;
        using Self  = ActivationFactory;
        using TMat  = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
        using TMap  = Eigen::Map< TMat >;

        using TUnary = std::function< TReal( const TReal&, bool ) >;
        using TFunction = std::function< void( TMap&, const TMap&, bool ) >;

      public:
        static TFunction Get( const std::string& a )
          {
            std::string n = a;
            std::transform(
              n.begin( ), n.end( ), n.begin( ),
              []( const unsigned char& c ) -> unsigned char
              {
                return( std::tolower( c ) );
              }
              );
            if( n == "softmax" )
            {
              return( []( TMap& A, const TMap& Z, bool d ) -> void {} );
            }
            else
            {
              if( n == "relu" )
                return(
                  []( TMap& A, const TMap& Z, bool d ) -> void
                  {
                    A = Z.unaryExpr(
                      [&d]( const TReal& z ) -> TReal
                      {
                        if( d )
                          return( ( z < TReal( 0 ) )? TReal( 0 ): TReal( 1 ) );
                        else
                          return( ( z < TReal( 0 ) )? TReal( 0 ): z );
                      }
                      );
                  }
                  );
              else if( n == "tanh" )
                return(
                  []( TMap& A, const TMap& Z, bool d ) -> void
                  {
                    A = Z.unaryExpr(
                      [&d]( const TReal& z ) -> TReal
                      {
                        TReal a = std::tanh( z );
                        if( d )
                          return( TReal( 1 ) - ( a * a ) );
                        else
                          return( a );
                      }
                      );
                  }
                  );
              else if( n == "sigmoid" )
                return(
                  []( TMap& A, const TMap& Z, bool d ) -> void
                  {
                    A = Z.unaryExpr(
                      [&d]( const TReal& z ) -> TReal
                      {
                        static const TReal _0  = TReal( 0 );
                        static const TReal _1  = TReal( 1 );
                        static const TReal _M  = std::numeric_limits< TReal >::max( );
                        static const TReal _L  = std::log( _M ) / TReal( 2 );

                        TReal s;
                        if     ( z >  _L ) s = _1;
                        else if( z < -_L ) s = _0;
                        else               s = _1 / ( _1 + std::exp( -z ) );

                        return( s * ( ( d )? ( _1 - s ): _1 ) );
                      }
                      );
                  }
                  );
              else // if( n == "identity" )
                return(
                  []( TMap& A, const TMap& Z, bool d ) -> void
                  {
                    A = Z.unaryExpr(
                      [&d]( const TReal& z ) -> TReal
                      {
                        return( ( d )? TReal( 1 ): z );
                      }
                      );
                  }
                  );
            } // end if
          }
      };
      

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

        using TActivationFactory
        =
          ivqML::Model::NeuralNetwork::ActivationFactory< TReal >;
        using TActivation = typename TActivationFactory::TFunction;

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
            this->m_F.clear( );

            this->m_N.push_back( i );
            this->m_N.push_back( o );
            this->m_F.push_back( a );
          }
        void set_input_layer(
          const TNatural& i, const TNatural& o, const std::string& a
          )
          {
            this->set_input_layer( i, o, TActivationFactory::Get( a ) );
          }
        void add_layer( const TNatural& o, TActivation a )
          {
            this->m_N.push_back( o );
            this->m_F.push_back( a );
          }
        void add_layer( const TNatural& o, const std::string& a )
          {
            this->add_layer( o, TActivationFactory::Get( a ) );
          }
        TNatural number_of_layers( ) const
          {
            return( this->m_W.size( ) );
          }
        const TNatural& input_size( const TNatural& l = 0 ) const
          {
            static const TNatural _0 = TNatural( 0 );
            if( l < this->m_N.size( ) )
              return( this->m_N[ l ] );
            else
              return( _0 );
          }
        const TNatural& output_size( const TNatural& l = 0 ) const
          {
            static const TNatural _0 = TNatural( 0 );
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
                  return( _0 );
              }
            }
            else
              return( _0 );
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

            // Init some random parameters
            std::random_device rd;
            std::mt19937 rg( rd( ) );
            std::uniform_real_distribution< TReal > rdis(
              std::numeric_limits< TReal >::epsilon( ),
              TReal( 1 )
              );
            std::transform(
              this->m_P, this->m_P + this->m_S, this->m_P,
              [&]( const TReal& v ) -> TReal
              {
                return( ( TReal( 2 ) * rdis( rg ) ) - TReal( 1 ) );
              }
              );

          }

        template< class _TA >
        auto operator()( const Eigen::EigenBase< _TA >& bX ) const
          {
            auto X = bX.derived( ).template cast< TReal >( );
            TNatural L = this->number_of_layers( );
            TNatural M = X.rows( );
            TNatural mN
              =
              *( std::max_element( this->m_N.begin( ), this->m_N.end( ) ) );
            TReal* Ab
              =
              reinterpret_cast< TReal* >(
                std::calloc( ( mN << 1 ) * M, sizeof( TReal ) )
                );
            TReal* Zb = Ab + ( mN * M );

            TMatMap( Ab, M, this->m_N[ 0 ] ) = X;
            for( TNatural l = 0; l < L; ++l )
            {
              TNatural i = this->m_N[ l ];
              TNatural o = this->m_N[ l + 1 ];

              TMatMap Z( Zb, M, o );

              std::cout << "................." << std::endl;
              std::cout << TMatMap( Ab, M, i ) << std::endl;
              std::cout << "................." << std::endl;

              Z = ( TMatMap( Ab, M, i ) * this->m_W[ l ] ) + this->m_B[ l ];
              std::cout << Z << std::endl;
              std::cout << "+++++++++++++++++" << std::endl;

              TMatMap A( Ab, M, o );
              this->m_F[ l ]( A, Z, false );
              std::cout << A << std::endl;
              std::cout << "*****************" << std::endl;
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
            this->Superclass::_to_stream( o );
          }

      protected:
        std::vector< TNatural >    m_N;
        std::vector< TMatMap >     m_W;
        std::vector< TRowMap >     m_B;
        std::vector< TActivation > m_F;
      };
    } // end namespace
  } // end namespace
} // end namespace

// TODO: #include <ivqML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __ivqML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
