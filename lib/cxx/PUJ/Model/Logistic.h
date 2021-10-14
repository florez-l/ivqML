// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Logistic__h__
#define __PUJ__Logistic__h__

#include <PUJ/Model/Linear.h>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
    class Logistic
      : public PUJ::Model::Linear< _TScalar, _TTraits >
    {
    public:
      using Superclass = PUJ::Model::Linear< _TScalar, _TTraits >;
      PUJ_TraitsMacro( Logistic );

    protected:
      using _TBaseCost = typename Superclass::Cost;

    public:
      Logistic( );
      Logistic( const TRow& t );

      template< class _TNumber, class ... _TArgs >
      Logistic( const _TNumber& v, _TArgs... args )
        : Superclass( v, args... )
        {
        }
        /* TODO
           : m_W( nullptr )
           {
           if( sizeof...( args ) > 0 )
           {
           Self n( args... );
           this->SetParameters( TRow::Zero( sizeof...( args ) + 1 ) );
           this->SetWeights( n.GetParameters( ) );
           this->SetBias( TScalar( v ) );
           }
           else
           {
           TRow p( 1 );
           p( 0, 0 ) = TScalar( v );
           this->SetParameters( p );
           } // end if
           }
        */

      virtual ~Logistic( ) = default;

      virtual TScalar operator()( const TRow& x ) const override;
      virtual TCol operator()( const TMatrix& x ) const override;

    public:

      /**
       */
      class Cost
        : public _TBaseCost
      {
      public:
        Cost(
          Self* model, const TMatrix& X, const TCol& y,
          unsigned int batch_size = 0
          );
        virtual ~Cost( ) = default;

        virtual TScalar operator()(
          unsigned int i, TRow* g = nullptr
          ) const override;

      protected:
        std::vector< std::vector< unsigned long long > > m_Zeros;
        std::vector< std::vector< unsigned long long > > m_Ones;
      };
    };


    /**
       template< class _TScalar, class _TTraits = PUJ::Traits< _TScalar > >
       class Logistic
       : public Linear< _TScalar, _TTraits >
       {
       public:
       PUJ_TraitsMacro( Logistic );
       using Superclass = PUJ::Model::Linear< _TScalar, _TTraits >;

       public:
       Logistic( const TRow& w, const TScalar& b );
       virtual ~Logistic( ) = default;

       virtual TScalar operator()( const TRow& x ) const override;
       virtual TCol operator()( const TMatrix& x ) const override;

       public:
       class Cost
       {
       public:
       Cost( const TMatrix& X, const TCol& y );
       virtual ~Cost( ) = default;

       TScalar operator()( const TRow& t, TRow* g = nullptr ) const;

       protected:
       TMatrix m_X;
       TCol    m_y;
       TRow    m_Xy;
       TScalar m_uy;
       };
       };
    */

  } // end namespace
} // end namespace

#endif // __PUJ__Logistic__h__

// eof - $RCSfile$
